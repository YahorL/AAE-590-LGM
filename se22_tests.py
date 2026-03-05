import pytest

import numpy as np
from so2 import so2_wedge, so2_exp
from se22 import (
    se22_compose,
    se22_inverse,
    se22_wedge,
    se22_vee,
    se22_exp,
    se22_log,
    se22_Ad,
    se22_ad,
)


class InvMat:
    def __init__(self, array):
        self.array = np.array(array, dtype=float)

    def inv(self):
        return InvMat(np.linalg.inv(self.array))

    def __matmul__(self, other):
        if isinstance(other, InvMat):
            return InvMat(self.array @ other.array)
        return self.array @ other

    def __neg__(self):
        return InvMat(-self.array)


@pytest.mark.parametrize("A1,A2,b1,b2,c1,c2", [
    (
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    ),
    (
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        np.array([[1.0, -1.0], [2.0, 3.0]]),
        np.array([1.0, -2.0]),
        np.array([3.0, 4.0]),
        np.array([-1.0, 0.5]),
        np.array([2.0, -3.0]),
    ),
    (
        np.array([[0.8, 0.3], [-0.2, 1.5]]),
        np.array([[1.2, -0.4], [0.7, 0.9]]),
        np.array([-1.5, 0.5]),
        np.array([2.0, -3.0]),
        np.array([0.4, -2.1]),
        np.array([1.7, 0.8]),
    ),
])
def test_se22_compose(A1, A2, b1, b2, c1, c2):
    X1 = (InvMat(A1), b1, c1)
    X2 = (InvMat(A2), b2, c2)
    A_expected = A1 @ A2
    b_expected = A1 @ b2 + b1
    c_expected = A1 @ c2 + c1

    A_result, b_result, c_result = se22_compose(X1, X2)
    np.testing.assert_array_almost_equal(A_result.array, A_expected)
    np.testing.assert_array_almost_equal(b_result, b_expected)
    np.testing.assert_array_almost_equal(c_result, c_expected)


@pytest.mark.parametrize("A,b,c", [
    (
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    ),
    (
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        np.array([1.0, -1.0]),
        np.array([-2.0, 3.5]),
    ),
    (
        np.array([[0.8, 0.3], [-0.2, 1.5]]),
        np.array([-1.5, 0.5]),
        np.array([0.4, -2.1]),
    ),
])
def test_se22_inverse(A, b, c):
    X = (InvMat(A), b, c)
    A_inv_expected = np.linalg.inv(A)
    b_inv_expected = -(A_inv_expected @ b)
    c_inv_expected = -(A_inv_expected @ c)

    A_inv_result, b_inv_result, c_inv_result = se22_inverse(X)
    np.testing.assert_array_almost_equal(A_inv_result.array, A_inv_expected)
    np.testing.assert_array_almost_equal(b_inv_result, b_inv_expected)
    np.testing.assert_array_almost_equal(c_inv_result, c_inv_expected)


@pytest.mark.parametrize("A,b,c", [
    (
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    ),
    (
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        np.array([1.0, -1.0]),
        np.array([-2.0, 3.5]),
    ),
    (
        np.array([[0.8, 0.3], [-0.2, 1.5]]),
        np.array([-1.5, 0.5]),
        np.array([0.4, -2.1]),
    ),
])
def test_se22_compose_with_inverse(A, b, c):
    X = (InvMat(A), b, c)
    X_inv = se22_inverse(X)
    A_result, b_result, c_result = se22_compose(X, X_inv)

    np.testing.assert_array_almost_equal(A_result.array, np.eye(2))
    np.testing.assert_array_almost_equal(b_result, np.zeros(2))
    np.testing.assert_array_almost_equal(c_result, np.zeros(2))


@pytest.mark.parametrize("xi,expected", [
    (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
    (
        np.array([1.0, -2.0, 3.0, -4.0, 0.5]),
        np.array([
            [0.0, -0.5, 1.0, 3.0],
            [0.5, 0.0, -2.0, -4.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
    (
        np.array([-3.0, 4.0, -1.0, 2.0, -1.2]),
        np.array([
            [0.0, 1.2, -3.0, -1.0],
            [-1.2, 0.0, 4.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
])
def test_se22_wedge(xi, expected):
    result = se22_wedge(xi)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("xi", [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([1.0, -2.0, 3.0, -4.0, 0.5]),
    np.array([-3.0, 4.0, -1.0, 2.0, -1.2]),
])
def test_se22_vee_wedge_roundtrip(xi):
    Xi = se22_wedge(xi)
    result = se22_vee(Xi)
    np.testing.assert_array_almost_equal(result, xi)


@pytest.mark.parametrize("xi", [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([2.0, -0.5, -1.0, 1.5, np.pi / 6]),
    np.array([-1.0, 1.5, 0.7, -2.2, -np.pi / 4]),
])
def test_se22_wedge_blocks(xi):
    a1, a2, b1, b2, theta = xi
    Xi = se22_wedge(xi)
    np.testing.assert_array_almost_equal(Xi[:2, :2], so2_wedge(theta))
    np.testing.assert_array_almost_equal(Xi[:2, 2], np.array([a1, a2]))
    np.testing.assert_array_almost_equal(Xi[:2, 3], np.array([b1, b2]))
    np.testing.assert_array_almost_equal(Xi[2:, :2], np.zeros((2, 2)))
    np.testing.assert_array_almost_equal(Xi[2:, 2:], np.eye(2))


def _se22_exp_expected(xi):
    theta = xi[4]
    K = np.array([[0.0, -1.0], [1.0, 0.0]])
    if np.abs(theta) < 1e-8:
        V = np.eye(2) + 0.5 * theta * K
    else:
        V = (np.sin(theta) / theta) * np.eye(2) + ((1.0 - np.cos(theta)) / theta) * K
    a = V @ np.array([xi[0], xi[1]])
    b = V @ np.array([xi[2], xi[3]])
    return np.block([
        [so2_exp(theta), a.reshape(2, 1), b.reshape(2, 1)],
        [np.zeros((1, 2)), np.array([[1.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[1.0]])],
    ])


@pytest.mark.parametrize("xi", [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([1.5, -2.0, 0.7, -1.3, 0.0]),
    np.array([-3.0, 4.5, -2.2, 1.1, 0.0]),
])
def test_se22_exp_zero_theta(xi):
    a1, a2, b1, b2, _ = xi
    result = se22_exp(xi)
    expected = np.block([
        [np.eye(2), np.array([[a1], [a2]]), np.array([[b1], [b2]])],
        [np.zeros((1, 2)), np.array([[1.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[1.0]])],
    ])
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("xi", [
    np.array([1.0, -0.5, 2.0, -1.0, 1e-10]),
    np.array([-2.0, 3.0, -1.2, 0.8, -1e-9]),
    np.array([0.3, -1.2, 1.7, -2.5, 1e-6]),
])
def test_se22_exp_near_zero(xi):
    result = se22_exp(xi)
    expected = _se22_exp_expected(xi)
    np.testing.assert_array_almost_equal(result, expected, decimal=7)


@pytest.mark.parametrize("xi", [
    np.array([1.0, 2.0, -1.0, 0.5, 0.2]),
    np.array([-1.5, 0.4, 2.2, -0.7, -0.7]),
    np.array([0.0, -2.2, 1.3, -1.1, 1.3]),
])
def test_se22_exp_blocks(xi):
    _, _, _, _, theta = xi
    result = se22_exp(xi)
    np.testing.assert_array_almost_equal(result[:2, :2], so2_exp(theta))
    np.testing.assert_array_almost_equal(result[2:, :2], np.zeros((2, 2)))
    np.testing.assert_array_almost_equal(result[2:, 2:], np.eye(2))


@pytest.mark.parametrize("xi", [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([1.0, -2.0, 0.5, -0.3, 1e-10]),
    np.array([-1.5, 0.5, 2.1, -0.8, -1e-9]),
    np.array([0.3, -0.7, -1.2, 1.4, 1e-6]),
])
def test_se22_log_near_zero(xi):
    X = se22_exp(xi)
    result = se22_log(X)
    np.testing.assert_array_almost_equal(result, xi, decimal=7)


@pytest.mark.parametrize("xi", [
    np.array([1.2, -0.8, 2.4, -1.7, 0.4]),
    np.array([-2.5, 3.1, -1.9, 0.6, -1.1]),
    np.array([0.0, 2.2, 1.1, -2.4, 1.7]),
])
def test_se22_log_regular_values(xi):
    X = se22_exp(xi)
    result = se22_log(X)
    np.testing.assert_array_almost_equal(result, xi, decimal=7)


def test_se22_exp_log_random():
    rng = np.random.default_rng(3)
    for _ in range(200):
        xi = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-1.0, 1.0),
        ])
        X = se22_exp(xi)
        X_roundtrip = se22_exp(se22_log(X))
        np.testing.assert_array_almost_equal(X_roundtrip, X, decimal=7)


def test_se22_exp_log_roundtrip_three_random_X():
    rng = np.random.default_rng(42)
    for _ in range(3):
        xi = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-1.0, 1.0),
        ])
        X = se22_exp(xi)
        X_roundtrip = se22_exp(se22_log(X))
        np.testing.assert_array_almost_equal(X_roundtrip, X, decimal=7)


def _se22_hat(xi):
    return np.block([
        [so2_wedge(xi[4]), np.array([[xi[0]], [xi[1]]]), np.array([[xi[2]], [xi[3]]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[0.0]])],
    ])


def _se22_vee_hat(Xi):
    return np.array([Xi[0, 2], Xi[1, 2], Xi[0, 3], Xi[1, 3], Xi[1, 0]])


def _se22_tuple_to_matrix(X):
    R, v, p = X
    return np.block([
        [R, v.reshape(2, 1), p.reshape(2, 1)],
        [np.zeros((1, 2)), np.array([[1.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[1.0]])],
    ])


def _xi_to_se22_tuple(xi):
    R = so2_exp(xi[4])
    v = np.array([xi[0], xi[1]])
    p = np.array([xi[2], xi[3]])
    return R, v, p


def test_se22_Ad_composition_random():
    rng = np.random.default_rng(11)
    for _ in range(20):
        xi1 = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-np.pi, np.pi),
        ])
        xi2 = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-np.pi, np.pi),
        ])
        X1 = _xi_to_se22_tuple(xi1)
        X2 = _xi_to_se22_tuple(xi2)
        X12 = se22_compose(X1, X2)
        Ad_12 = se22_Ad(X12)
        Ad_1 = se22_Ad(X1)
        Ad_2 = se22_Ad(X2)
        np.testing.assert_array_almost_equal(Ad_12, Ad_1 @ Ad_2, decimal=7)


def test_se22_ad_bracket_definition_random():
    rng = np.random.default_rng(12)
    for _ in range(20):
        xi1 = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        ])
        xi2 = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        ])
        bracket_hat = _se22_hat(xi1) @ _se22_hat(xi2) - _se22_hat(xi2) @ _se22_hat(xi1)
        bracket = _se22_vee_hat(bracket_hat)
        ad_action = se22_ad(xi1) @ xi2
        np.testing.assert_array_almost_equal(bracket, ad_action, decimal=7)


def test_se22_Ad_definition_conjugation_random():
    rng = np.random.default_rng(13)
    for _ in range(20):
        xi_X = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-np.pi, np.pi),
        ])
        eta = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        ])
        X_tuple = _xi_to_se22_tuple(xi_X)
        X_mat = _se22_tuple_to_matrix(X_tuple)
        lhs = _se22_vee_hat(X_mat @ _se22_hat(eta) @ np.linalg.inv(X_mat))
        rhs = se22_Ad(X_tuple) @ eta
        np.testing.assert_array_almost_equal(lhs, rhs, decimal=7)
