import pytest

import numpy as np
from so2 import so2_exp, so2_wedge, so2_log
from se2 import se2_compose, se2_inverse, se2_wedge, se2_vee, se2_exp, se2_log, se2_Ad

@pytest.mark.parametrize("theta1,theta2,t1,t2", [
    (0.0, 0.0, np.array([0.0, 0.0]), np.array([0.0, 0.0])),
    (np.pi / 4, -np.pi / 6, np.array([1.0, -2.0]), np.array([3.0, 4.0])),
    (-np.pi / 2, np.pi / 3, np.array([-1.5, 0.5]), np.array([2.0, -3.0])),
])
def test_se2_compose(theta1, theta2, t1, t2):
    R1 = so2_exp(theta1)
    R2 = so2_exp(theta2)
    X1 = (t1, R1)
    X2 = (t2, R2)
    t_expected = t1 + R1 @ t2
    R_expected = R1 @ R2
    t_result, R_result = se2_compose(X1, X2)
    np.testing.assert_array_almost_equal(t_result, t_expected)
    np.testing.assert_array_almost_equal(R_result, R_expected)


@pytest.mark.parametrize("theta,t", [
    (0.0, np.array([0.0, 0.0])),
    (np.pi / 3, np.array([1.0, -1.0])),
    (-np.pi / 2, np.array([-2.0, 3.5])),
])
def test_se2_inverse(theta, t):
    R = so2_exp(theta)
    X = (t, R)
    t_inv_expected = -R.T @ t
    R_inv_expected = R.T
    t_inv_result, R_inv_result = se2_inverse(X)
    np.testing.assert_array_almost_equal(t_inv_result, t_inv_expected)
    np.testing.assert_array_almost_equal(R_inv_result, R_inv_expected)


@pytest.mark.parametrize("theta,t", [
    (0.0, np.array([0.0, 0.0])),
    (np.pi / 6, np.array([2.0, -0.5])),
    (-np.pi / 4, np.array([-1.0, 1.5])),
])
def test_se2_compose_with_inverse(theta, t):
    R = so2_exp(theta)
    X = (t, R)
    X_inv = se2_inverse(X)
    t_result, R_result = se2_compose(X, X_inv)
    np.testing.assert_array_almost_equal(t_result, np.zeros(2))
    np.testing.assert_array_almost_equal(R_result, np.eye(2))


@pytest.mark.parametrize("xi,expected", [
    (np.array([0.0, 0.0, 0.0]), np.zeros((3, 3))),
    (np.array([1.0, -2.0, 0.5]), np.array([
        [0.0, -0.5, 1.0],
        [0.5,  0.0, -2.0],
        [0.0,  0.0,  0.0],
    ])),
    (np.array([-3.0, 4.0, -1.2]), np.array([
        [0.0,  1.2, -3.0],
        [-1.2, 0.0,  4.0],
        [0.0,  0.0,  0.0],
    ])),
])
def test_se2_wedge(xi, expected):
    result = se2_wedge(xi)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("xi", [
    (np.array([0.0, 0.0, 0.0])),
    (np.array([1.0, -2.0, 0.5])),
    (np.array([-3.0, 4.0, -1.2])),
])
def test_se2_vee_wedge_roundtrip(xi):
    Xi = se2_wedge(xi)
    result = se2_vee(Xi)
    np.testing.assert_array_almost_equal(result, xi)


@pytest.mark.parametrize("xi", [
    (np.array([0.0, 0.0, 0.0])),
    (np.array([2.0, -1.0, np.pi / 4])),
    (np.array([-3.5, 4.2, -np.pi / 3])),
])
def test_se2_wedge_blocks(xi):
    x, y, theta = xi
    Xi = se2_wedge(xi)
    np.testing.assert_array_almost_equal(Xi[:2, :2], so2_wedge(theta))
    np.testing.assert_array_almost_equal(Xi[:2, 2], np.array([x, y]))
    np.testing.assert_array_almost_equal(Xi[2, :], np.array([0.0, 0.0, 0.0]))


def _se2_exp_expected(xi):
    theta = xi[2]
    K = np.array([[0.0, -1.0], [1.0, 0.0]])
    if np.abs(theta) < 1e-8:
        V = np.eye(2) + 0.5 * theta * K
    else:
        V = (np.sin(theta) / theta) * np.eye(2) + ((1.0 - np.cos(theta)) / theta) * K
    t = V @ np.array([xi[0], xi[1]])
    return np.block([
        [so2_exp(theta), t.reshape(2, 1)],
        [np.zeros((1, 2)), np.array([[1.0]])],
    ])


@pytest.mark.parametrize("xi", [
    (np.array([0.0, 0.0, 0.0])),
    (np.array([1.5, -2.0, 0.0])),
    (np.array([-3.0, 4.5, 0.0])),
])
def test_se2_exp_zero_theta(xi):
    x, y, _ = xi
    result = se2_exp(xi)
    expected = np.block([
        [np.eye(2), np.array([[x], [y]])],
        [np.zeros((1, 2)), np.array([[1.0]])],
    ])
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("xi", [
    (np.array([1.0, -0.5, 1e-10])),
    (np.array([-2.0, 3.0, -1e-9])),
    (np.array([0.3, -1.2, 1e-6])),
])
def test_se2_exp_near_zero(xi):
    result = se2_exp(xi)
    expected = _se2_exp_expected(xi)
    np.testing.assert_array_almost_equal(result, expected, decimal=7)


@pytest.mark.parametrize("xi", [
    (np.array([1.0, 2.0, 0.2])),
    (np.array([-1.5, 0.4, -0.7])),
    (np.array([0.0, -2.2, 1.3])),
])
def test_se2_exp_blocks(xi):
    _, _, theta = xi
    result = se2_exp(xi)
    np.testing.assert_array_almost_equal(result[:2, :2], so2_exp(theta))
    np.testing.assert_array_almost_equal(result[2, :], np.array([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("xi", [
    (np.array([0.0, 0.0, 0.0])),
    (np.array([1.0, -2.0, 1e-10])),
    (np.array([-1.5, 0.5, -1e-9])),
    (np.array([0.3, -0.7, 1e-6])),
])
def test_se2_log_near_zero(xi):
    X = se2_exp(xi)
    result = se2_log(X)
    np.testing.assert_array_almost_equal(result, xi, decimal=7)


@pytest.mark.parametrize("xi", [
    (np.array([1.2, -0.8, 0.4])),
    (np.array([-2.5, 3.1, -1.1])),
    (np.array([0.0, 2.2, 1.7])),
])
def test_se2_log_regular_values(xi):
    X = se2_exp(xi)
    result = se2_log(X)
    np.testing.assert_array_almost_equal(result, xi, decimal=7)


def test_se2_exp_log_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        xi = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-1.0, 1.0),
        ])
        X = se2_exp(xi)
        X_roundtrip = se2_exp(se2_log(X))
        np.testing.assert_array_almost_equal(X_roundtrip, X, decimal=7)


def _xi_to_tuple(xi):
    t = np.array([xi[0], xi[1]])
    R = so2_exp(xi[2])
    return t, R


def _tuple_to_xi(X):
    t, R = X
    theta = so2_log(R)
    return np.array([t[0], t[1], theta])


def test_se2_adjoint_composition_random():
    rng = np.random.default_rng(1)
    for _ in range(50):
        xi1 = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-np.pi, np.pi),
        ])
        xi2 = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-np.pi, np.pi),
        ])
        X1 = _xi_to_tuple(xi1)
        X2 = _xi_to_tuple(xi2)
        X12 = se2_compose(X1, X2)
        xi12 = _tuple_to_xi(X12)
        Ad_12 = se2_Ad(xi12)
        Ad_1 = se2_Ad(xi1)
        Ad_2 = se2_Ad(xi2)
        np.testing.assert_array_almost_equal(Ad_12, Ad_1 @ Ad_2, decimal=7)


def test_se2_adjoint_inverse_random():
    rng = np.random.default_rng(2)
    for _ in range(50):
        xi = np.array([
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-np.pi, np.pi),
        ])
        X = _xi_to_tuple(xi)
        X_inv = se2_inverse(X)
        xi_inv = _tuple_to_xi(X_inv)
        Ad_inv = se2_Ad(xi_inv)
        Ad = se2_Ad(xi)
        np.testing.assert_array_almost_equal(Ad_inv, np.linalg.inv(Ad), decimal=7)