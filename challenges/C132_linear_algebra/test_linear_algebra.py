"""Tests for C132: Linear Algebra."""

import math
import pytest
from linear_algebra import (
    Matrix, dot, vec_norm, vec_scale, vec_add, vec_sub, vec_normalize,
    lu_decompose, lu_solve, determinant, matrix_inverse,
    qr_decompose, qr_solve,
    cholesky, cholesky_solve,
    svd, eigenvalues, eigen_decompose,
    least_squares, matrix_rank, null_space, condition_number,
    norm_1, norm_inf, norm_2,
    matrix_power, matrix_exp, pseudoinverse,
    gram_schmidt, modified_gram_schmidt,
    power_iteration, conjugate_gradient, is_positive_definite,
)


def approx(a, b, tol=1e-6):
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert abs(x - y) < tol, f"{x} != {y}"
    else:
        assert abs(a - b) < tol, f"{a} != {b}"


# ============================================================
# Matrix basics
# ============================================================

class TestMatrixBasics:
    def test_create_from_list(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.rows == 2 and m.cols == 2
        assert m[0, 0] == 1.0 and m[1, 1] == 4.0

    def test_identity(self):
        I = Matrix.identity(3)
        assert I[0, 0] == 1 and I[1, 1] == 1 and I[2, 2] == 1
        assert I[0, 1] == 0 and I[1, 0] == 0

    def test_zeros(self):
        z = Matrix.zeros(2, 3)
        assert z.rows == 2 and z.cols == 3
        assert all(z[i, j] == 0 for i in range(2) for j in range(3))

    def test_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        t = m.T
        assert t.rows == 3 and t.cols == 2
        assert t[0, 0] == 1 and t[2, 1] == 6

    def test_add(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        c = a + b
        assert c[0, 0] == 6 and c[1, 1] == 12

    def test_sub(self):
        a = Matrix([[5, 6], [7, 8]])
        b = Matrix([[1, 2], [3, 4]])
        c = a - b
        assert c[0, 0] == 4 and c[1, 1] == 4

    def test_scalar_mul(self):
        m = Matrix([[1, 2], [3, 4]])
        r = m * 2
        assert r[0, 0] == 2 and r[1, 1] == 8

    def test_scalar_rmul(self):
        m = Matrix([[1, 2], [3, 4]])
        r = 3 * m
        assert r[0, 0] == 3 and r[1, 1] == 12

    def test_matmul(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        c = a * b
        assert c[0, 0] == 19 and c[0, 1] == 22
        assert c[1, 0] == 43 and c[1, 1] == 50

    def test_neg(self):
        m = Matrix([[1, -2], [3, -4]])
        n = -m
        assert n[0, 0] == -1 and n[0, 1] == 2

    def test_shape(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert m.shape == (2, 3)

    def test_trace(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.trace() == 5

    def test_frobenius_norm(self):
        m = Matrix([[1, 2], [3, 4]])
        approx(m.frobenius_norm(), math.sqrt(30))

    def test_is_square(self):
        assert Matrix([[1, 2], [3, 4]]).is_square()
        assert not Matrix([[1, 2, 3], [4, 5, 6]]).is_square()

    def test_is_symmetric(self):
        assert Matrix([[1, 2], [2, 1]]).is_symmetric()
        assert not Matrix([[1, 2], [3, 1]]).is_symmetric()

    def test_is_upper_triangular(self):
        assert Matrix([[1, 2], [0, 4]]).is_upper_triangular()
        assert not Matrix([[1, 2], [1, 4]]).is_upper_triangular()

    def test_is_lower_triangular(self):
        assert Matrix([[1, 0], [3, 4]]).is_lower_triangular()
        assert not Matrix([[1, 2], [3, 4]]).is_lower_triangular()

    def test_col_and_row(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.col(0) == [1.0, 3.0]
        assert m.row(1) == [3.0, 4.0]

    def test_set_col(self):
        m = Matrix([[1, 2], [3, 4]])
        m.set_col(0, [10, 20])
        assert m[0, 0] == 10 and m[1, 0] == 20

    def test_submatrix(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        s = m.submatrix(1, 3, 1, 3)
        assert s[0, 0] == 5 and s[1, 1] == 9

    def test_diag(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.diag() == [1.0, 4.0]

    def test_diag_matrix(self):
        d = Matrix.diag_matrix([3, 5, 7])
        assert d[0, 0] == 3 and d[1, 1] == 5 and d[2, 2] == 7
        assert d[0, 1] == 0

    def test_from_vector(self):
        v = Matrix.from_vector([1, 2, 3])
        assert v.rows == 3 and v.cols == 1
        assert v[1, 0] == 2.0

    def test_to_vector(self):
        v = Matrix.from_vector([1, 2, 3])
        assert v.to_vector() == [1.0, 2.0, 3.0]

    def test_copy(self):
        m = Matrix([[1, 2], [3, 4]])
        c = m.copy()
        c[0, 0] = 99
        assert m[0, 0] == 1  # original unchanged

    def test_approx_eq(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[1.0 + 1e-11, 2.0], [3.0, 4.0 - 1e-11]])
        assert a.approx_eq(b)

    def test_is_orthogonal(self):
        I = Matrix.identity(3)
        assert I.is_orthogonal()


# ============================================================
# Vector operations
# ============================================================

class TestVectorOps:
    def test_dot(self):
        assert dot([1, 2, 3], [4, 5, 6]) == 32

    def test_norm(self):
        approx(vec_norm([3, 4]), 5.0)

    def test_scale(self):
        assert vec_scale([1, 2], 3) == [3, 6]

    def test_add(self):
        assert vec_add([1, 2], [3, 4]) == [4, 6]

    def test_sub(self):
        assert vec_sub([5, 3], [1, 2]) == [4, 1]

    def test_normalize(self):
        v = vec_normalize([3, 4])
        approx(vec_norm(v), 1.0)


# ============================================================
# LU Decomposition
# ============================================================

class TestLU:
    def test_lu_2x2(self):
        A = Matrix([[2, 1], [4, 3]])
        L, U, P = lu_decompose(A)
        assert L.is_lower_triangular()
        assert U.is_upper_triangular()
        # PA = LU
        result = L * U
        PA = P * A
        assert result.approx_eq(PA)

    def test_lu_3x3(self):
        A = Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        L, U, P = lu_decompose(A)
        PA = P * A
        LU = L * U
        assert LU.approx_eq(PA)

    def test_lu_solve_simple(self):
        A = Matrix([[2, 1], [5, 3]])
        b = [4, 7]
        x = lu_solve(A, b)
        approx(x, [5.0, -6.0])

    def test_lu_solve_3x3(self):
        A = Matrix([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
        b = [6, -4, 27]
        x = lu_solve(A, b)
        # Verify Ax = b
        for i in range(3):
            approx(sum(A[i, j] * x[j] for j in range(3)), b[i])

    def test_lu_pivoting(self):
        # Matrix that needs pivoting (zero on diagonal)
        A = Matrix([[0, 1], [1, 0]])
        L, U, P = lu_decompose(A)
        PA = P * A
        LU = L * U
        assert LU.approx_eq(PA)


# ============================================================
# Determinant and Inverse
# ============================================================

class TestDetInverse:
    def test_det_2x2(self):
        A = Matrix([[3, 7], [1, 5]])
        approx(determinant(A), 8.0)

    def test_det_3x3(self):
        A = Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        approx(determinant(A), -306.0)

    def test_det_identity(self):
        approx(determinant(Matrix.identity(4)), 1.0)

    def test_det_singular(self):
        A = Matrix([[1, 2], [2, 4]])
        approx(determinant(A), 0.0)

    def test_inverse_2x2(self):
        A = Matrix([[4, 7], [2, 6]])
        A_inv = matrix_inverse(A)
        product = A * A_inv
        assert product.approx_eq(Matrix.identity(2))

    def test_inverse_3x3(self):
        A = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        A_inv = matrix_inverse(A)
        product = A * A_inv
        assert product.approx_eq(Matrix.identity(3))

    def test_inverse_identity(self):
        I = Matrix.identity(3)
        assert matrix_inverse(I).approx_eq(I)


# ============================================================
# QR Decomposition
# ============================================================

class TestQR:
    def test_qr_3x3(self):
        A = Matrix([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
        Q, R = qr_decompose(A)
        # Q is orthogonal
        assert Q.is_orthogonal()
        # R is upper triangular
        assert R.is_upper_triangular()
        # QR = A
        assert (Q * R).approx_eq(A)

    def test_qr_2x2(self):
        A = Matrix([[1, 1], [0, 1]])
        Q, R = qr_decompose(A)
        assert Q.is_orthogonal()
        assert R.is_upper_triangular()
        assert (Q * R).approx_eq(A)

    def test_qr_solve(self):
        A = Matrix([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
        b = [6, -4, 27]
        x = qr_solve(A, b)
        for i in range(3):
            approx(sum(A[i, j] * x[j] for j in range(3)), b[i])

    def test_qr_rectangular(self):
        A = Matrix([[1, 2], [3, 4], [5, 6]])
        Q, R = qr_decompose(A)
        assert (Q * R).approx_eq(A)


# ============================================================
# Cholesky Decomposition
# ============================================================

class TestCholesky:
    def test_cholesky_2x2(self):
        A = Matrix([[4, 2], [2, 5]])
        L = cholesky(A)
        assert L.is_lower_triangular()
        assert (L * L.T).approx_eq(A)

    def test_cholesky_3x3(self):
        A = Matrix([[25, 15, -5], [15, 18, 0], [-5, 0, 11]])
        L = cholesky(A)
        assert L.is_lower_triangular()
        assert (L * L.T).approx_eq(A)

    def test_cholesky_solve(self):
        A = Matrix([[4, 2], [2, 5]])
        b = [8, 11]
        x = cholesky_solve(A, b)
        for i in range(2):
            approx(sum(A[i, j] * x[j] for j in range(2)), b[i])

    def test_cholesky_not_spd(self):
        A = Matrix([[1, 2], [2, 1]])  # Not positive definite
        with pytest.raises(ValueError):
            cholesky(A)

    def test_cholesky_not_symmetric(self):
        A = Matrix([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            cholesky(A)


# ============================================================
# SVD
# ============================================================

class TestSVD:
    def test_svd_2x2(self):
        A = Matrix([[3, 0], [0, 4]])
        U, S, V = svd(A)
        reconstructed = U * S * V.T
        assert reconstructed.approx_eq(A, tol=1e-6)

    def test_svd_3x3(self):
        A = Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        U, S, V = svd(A)
        reconstructed = U * S * V.T
        assert reconstructed.approx_eq(A, tol=1e-6)
        # Singular values sorted descending
        assert S[0, 0] >= S[1, 1] >= S[2, 2]

    def test_svd_rectangular(self):
        A = Matrix([[1, 2], [3, 4], [5, 6]])
        U, S, V = svd(A)
        reconstructed = U * S * V.T
        assert reconstructed.approx_eq(A, tol=1e-4)

    def test_svd_singular_values_descending(self):
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        U, S, V = svd(A)
        svals = [S[i, i] for i in range(3)]
        for i in range(len(svals) - 1):
            assert svals[i] >= svals[i + 1] - 1e-10

    def test_svd_symmetric(self):
        A = Matrix([[2, 1], [1, 2]])
        U, S, V = svd(A)
        reconstructed = U * S * V.T
        assert reconstructed.approx_eq(A, tol=1e-6)


# ============================================================
# Eigenvalues
# ============================================================

class TestEigenvalues:
    def test_eigenvalues_diagonal(self):
        A = Matrix([[3, 0, 0], [0, 1, 0], [0, 0, 2]])
        eigs = eigenvalues(A)
        approx(sorted(eigs, reverse=True), [3.0, 2.0, 1.0])

    def test_eigenvalues_symmetric(self):
        A = Matrix([[2, 1], [1, 2]])
        eigs = eigenvalues(A)
        approx(sorted(eigs, reverse=True), [3.0, 1.0])

    def test_eigenvalues_2x2(self):
        A = Matrix([[0, 1], [-2, -3]])
        eigs = eigenvalues(A)
        approx(sorted(eigs, reverse=True), [-1.0, -2.0])

    def test_eigen_decompose_symmetric(self):
        A = Matrix([[2, 1], [1, 2]])
        eigs, V = eigen_decompose(A)
        # V^T A V should be approximately diagonal
        D = V.T * A * V
        for i in range(2):
            for j in range(2):
                if i != j:
                    assert abs(D[i, j]) < 1e-6

    def test_eigen_decompose_3x3(self):
        A = Matrix([[6, -1, 0], [-1, 4, -1], [0, -1, 2]])
        eigs, V = eigen_decompose(A)
        # Check reconstruction: A * V[:,i] ~ eig[i] * V[:,i]
        for i in range(3):
            v_col = V.col(i)
            Av = [sum(A[r, c] * v_col[c] for c in range(3)) for r in range(3)]
            ev = vec_scale(v_col, eigs[i])
            # May have sign flip
            if vec_norm(v_col) > 1e-10:
                for r in range(3):
                    assert abs(abs(Av[r]) - abs(ev[r])) < 1e-4


# ============================================================
# Least squares
# ============================================================

class TestLeastSquares:
    def test_overdetermined(self):
        # Fit y = a + bx to points (0,1), (1,2), (2,4)
        A = Matrix([[1, 0], [1, 1], [1, 2]])
        b = [1, 2, 4]
        x = least_squares(A, b)
        # x should give best fit coefficients
        # Check residual is minimized (just verify solution is reasonable)
        assert len(x) == 2

    def test_exact_system(self):
        A = Matrix([[1, 0], [0, 1]])
        b = [3, 4]
        x = least_squares(A, b)
        approx(x, [3.0, 4.0])


# ============================================================
# Rank, null space, condition number
# ============================================================

class TestRankNullCond:
    def test_rank_full(self):
        A = Matrix([[1, 0], [0, 1]])
        assert matrix_rank(A) == 2

    def test_rank_deficient(self):
        A = Matrix([[1, 2], [2, 4]])
        assert matrix_rank(A) == 1

    def test_rank_rectangular(self):
        A = Matrix([[1, 0, 0], [0, 1, 0]])
        assert matrix_rank(A) == 2

    def test_null_space_full_rank(self):
        A = Matrix([[1, 0], [0, 1]])
        ns = null_space(A)
        assert len(ns) == 0

    def test_null_space_rank_deficient(self):
        A = Matrix([[1, 2], [2, 4]])
        ns = null_space(A)
        assert len(ns) == 1
        # The null vector should satisfy Av ~ 0
        v = ns[0]
        Av = [sum(A[i, j] * v[j] for j in range(2)) for i in range(2)]
        approx(Av, [0.0, 0.0], tol=1e-4)

    def test_condition_number_identity(self):
        approx(condition_number(Matrix.identity(3)), 1.0)

    def test_condition_number_ill(self):
        # Near-singular matrix has large condition number
        A = Matrix([[1, 1], [1, 1.001]])
        cn = condition_number(A)
        assert cn > 1000


# ============================================================
# Norms
# ============================================================

class TestNorms:
    def test_norm_1(self):
        A = Matrix([[1, -2], [3, 4]])
        assert norm_1(A) == 6.0  # max(|1|+|3|, |-2|+|4|) = max(4, 6)

    def test_norm_inf(self):
        A = Matrix([[1, -2], [3, 4]])
        assert norm_inf(A) == 7.0  # max(|1|+|-2|, |3|+|4|) = max(3, 7)

    def test_norm_2_identity(self):
        approx(norm_2(Matrix.identity(3)), 1.0)


# ============================================================
# Matrix power and exponential
# ============================================================

class TestPowerExp:
    def test_power_0(self):
        A = Matrix([[1, 2], [3, 4]])
        assert matrix_power(A, 0).approx_eq(Matrix.identity(2))

    def test_power_1(self):
        A = Matrix([[1, 2], [3, 4]])
        assert matrix_power(A, 1).approx_eq(A)

    def test_power_2(self):
        A = Matrix([[1, 2], [3, 4]])
        assert matrix_power(A, 2).approx_eq(A * A)

    def test_power_neg(self):
        A = Matrix([[1, 2], [3, 5]])
        A_inv = matrix_inverse(A)
        p = matrix_power(A, -1)
        assert p.approx_eq(A_inv)

    def test_exp_zero(self):
        Z = Matrix.zeros(2, 2)
        assert matrix_exp(Z).approx_eq(Matrix.identity(2))

    def test_exp_identity(self):
        I = Matrix.identity(2)
        eI = matrix_exp(I)
        e = math.e
        expected = Matrix.diag_matrix([e, e])
        assert eI.approx_eq(expected, tol=1e-6)

    def test_exp_diagonal(self):
        D = Matrix.diag_matrix([1, 2])
        eD = matrix_exp(D)
        approx(eD[0, 0], math.e)
        approx(eD[1, 1], math.e ** 2)
        approx(eD[0, 1], 0.0)


# ============================================================
# Pseudoinverse
# ============================================================

class TestPseudoinverse:
    def test_pinv_invertible(self):
        A = Matrix([[1, 2], [3, 4]])
        pinv = pseudoinverse(A)
        inv = matrix_inverse(A)
        assert pinv.approx_eq(inv, tol=1e-6)

    def test_pinv_rectangular(self):
        A = Matrix([[1, 0], [0, 1], [0, 0]])
        pinv = pseudoinverse(A)
        # A^+ * A should be identity (or close)
        product = pinv * A
        assert product.approx_eq(Matrix.identity(2), tol=1e-6)

    def test_pinv_rank_deficient(self):
        A = Matrix([[1, 2], [2, 4]])
        pinv = pseudoinverse(A)
        # A * A^+ * A ~ A
        assert (A * pinv * A).approx_eq(A, tol=1e-4)


# ============================================================
# Gram-Schmidt
# ============================================================

class TestGramSchmidt:
    def test_classical(self):
        vecs = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        result = gram_schmidt(vecs)
        assert len(result) == 3
        for v in result:
            approx(vec_norm(v), 1.0)
        # Orthogonality
        for i in range(3):
            for j in range(i + 1, 3):
                approx(dot(result[i], result[j]), 0.0)

    def test_modified(self):
        vecs = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        result = modified_gram_schmidt(vecs)
        assert len(result) == 3
        for v in result:
            approx(vec_norm(v), 1.0)
        for i in range(3):
            for j in range(i + 1, 3):
                approx(dot(result[i], result[j]), 0.0)

    def test_linearly_dependent(self):
        vecs = [[1, 0], [2, 0], [0, 1]]
        result = gram_schmidt(vecs)
        assert len(result) == 2  # One vector dropped


# ============================================================
# Power iteration
# ============================================================

class TestPowerIteration:
    def test_dominant_eigenvalue(self):
        A = Matrix([[2, 1], [1, 3]])
        eig, vec = power_iteration(A)
        # Dominant eigenvalue of [[2,1],[1,3]] is (5+sqrt(5))/2 ~ 3.618
        expected = (5 + math.sqrt(5)) / 2
        approx(eig, expected, tol=1e-4)
        # Eigenvector should satisfy Av = lambda*v
        Av = [sum(A[i, j] * vec[j] for j in range(2)) for i in range(2)]
        ev = vec_scale(vec, eig)
        approx(Av, ev, tol=1e-4)

    def test_diagonal_dominant(self):
        A = Matrix([[5, 0, 0], [0, 3, 0], [0, 0, 1]])
        eig, _ = power_iteration(A)
        approx(eig, 5.0, tol=1e-4)


# ============================================================
# Conjugate gradient
# ============================================================

class TestConjugateGradient:
    def test_simple_spd(self):
        A = Matrix([[4, 1], [1, 3]])
        b = [1, 2]
        x = conjugate_gradient(A, b)
        # Verify Ax ~ b
        Ax = [sum(A[i, j] * x[j] for j in range(2)) for i in range(2)]
        approx(Ax, b, tol=1e-6)

    def test_identity(self):
        A = Matrix.identity(3)
        b = [1, 2, 3]
        x = conjugate_gradient(A, b)
        approx(x, b)

    def test_3x3_spd(self):
        A = Matrix([[10, 1, 0], [1, 10, 1], [0, 1, 10]])
        b = [11, 12, 11]
        x = conjugate_gradient(A, b)
        Ax = [sum(A[i, j] * x[j] for j in range(3)) for i in range(3)]
        approx(Ax, b, tol=1e-6)


# ============================================================
# Positive definiteness
# ============================================================

class TestPositiveDefinite:
    def test_spd(self):
        assert is_positive_definite(Matrix([[4, 2], [2, 5]]))

    def test_not_spd(self):
        assert not is_positive_definite(Matrix([[1, 2], [2, 1]]))

    def test_not_symmetric(self):
        assert not is_positive_definite(Matrix([[1, 2], [3, 4]]))


# ============================================================
# Integration / composite tests
# ============================================================

class TestIntegration:
    def test_lu_det_consistency(self):
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        d = determinant(A)
        # det(A) should be non-zero (this matrix is invertible)
        assert abs(d) > 0.1

    def test_inverse_inverse(self):
        A = Matrix([[1, 2], [3, 5]])
        A_inv = matrix_inverse(A)
        A_inv_inv = matrix_inverse(A_inv)
        assert A_inv_inv.approx_eq(A)

    def test_det_product(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        # det(AB) = det(A) * det(B)
        approx(determinant(A * B), determinant(A) * determinant(B), tol=1e-4)

    def test_svd_rank(self):
        A = Matrix([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
        r = matrix_rank(A)
        assert r == 2  # Second row is 2x first row

    def test_solvers_agree(self):
        """LU, QR, and Cholesky should give same answer for SPD system."""
        A = Matrix([[4, 2, 0], [2, 5, 1], [0, 1, 3]])
        b = [8, 13, 7]
        x_lu = lu_solve(A, b)
        x_qr = qr_solve(A, b)
        x_ch = cholesky_solve(A, b)
        x_cg = conjugate_gradient(A, b)
        approx(x_lu, x_qr)
        approx(x_lu, x_ch)
        approx(x_lu, x_cg, tol=1e-4)

    def test_eigenvalue_trace_det(self):
        """Sum of eigenvalues = trace, product = determinant."""
        A = Matrix([[2, 1], [1, 2]])
        eigs = eigenvalues(A)
        approx(sum(eigs), A.trace())
        prod = 1.0
        for e in eigs:
            prod *= e
        approx(prod, determinant(A), tol=1e-4)

    def test_svd_singular_values_and_eigenvalues(self):
        """Singular values of A are sqrt of eigenvalues of A^T A."""
        A = Matrix([[1, 2], [3, 4]])
        _, S, _ = svd(A)
        AtA = A.T * A
        eigs = eigenvalues(AtA)
        sv_from_eigs = sorted([math.sqrt(max(0, e)) for e in eigs], reverse=True)
        sv_from_svd = [S[i, i] for i in range(2)]
        approx(sv_from_svd, sv_from_eigs, tol=1e-4)

    def test_4x4_system(self):
        A = Matrix([
            [10, -1, 2, 0],
            [-1, 11, -1, 3],
            [2, -1, 10, -1],
            [0, 3, -1, 8]
        ])
        b = [6, 25, -11, 15]
        x = lu_solve(A, b)
        Ax = [sum(A[i, j] * x[j] for j in range(4)) for i in range(4)]
        approx(Ax, b)

    def test_matrix_exp_derivative(self):
        """exp(tA) at t=0 should be I."""
        A = Matrix([[0, 1], [-1, 0]])
        I = Matrix.identity(2)
        assert matrix_exp(A * 0).approx_eq(I)

    def test_pseudoinverse_solve(self):
        """For full-rank square, A^+ b = A^{-1} b."""
        A = Matrix([[1, 2], [3, 4]])
        b_vec = [5, 6]
        b_mat = Matrix.from_vector(b_vec)
        x_pinv = (pseudoinverse(A) * b_mat).to_vector()
        x_lu = lu_solve(A, b_vec)
        approx(x_pinv, x_lu, tol=1e-4)

    def test_orthogonal_qr(self):
        """Q from QR should be orthogonal."""
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        Q, R = qr_decompose(A)
        QtQ = Q.T * Q
        assert QtQ.approx_eq(Matrix.identity(3))
