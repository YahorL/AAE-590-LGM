import pytest

import numpy as np
from so3 import so3_exp, so3_wedge, so3_log
from se3 import se3_compose, se3_inverse, se3_wedge, se3_vee, se3_exp, se3_log, se3_Ad


def _xi_to_tuple(xi):
    t = np.array([xi[0], xi[1], xi[2]])
    R = so3_exp(np.array([xi[3], xi[4], xi[5]]))
    return t, R


def _tuple_to_matrix(X):
    t, R = X
    return np.block([
        [R, t.reshape(3, 1)],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])


@pytest.mark.parametrize("xi1,xi2", [
    (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    (np.array([1.0, -2.0, 0.5, 0.3, -0.4, 0.2]),
     np.array([3.0, 4.0, -1.0, -0.1, 0.6, 0.5])),
    (np.array([-1.5, 0.5, 2.0, np.pi/4, -np.pi/6, np.pi/3]),
     np.array([2.0, -3.0, 1.0, -np.pi/3, np.pi/5, -np.pi/4])),
])
def test_se3_compose(xi1, xi2):
    X1 = _xi_to_tuple(xi1)
    X2 = _xi_to_tuple(xi2)
    t_expected = X1[0] + X1[1] @ X2[0]
    R_expected = X1[1] @ X2[1]
    t_result, R_result = se3_compose(X1, X2)
    np.testing.assert_array_almost_equal(t_result, t_expected)
    np.testing.assert_array_almost_equal(R_result, R_expected)


@pytest.mark.parametrize("xi", [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([1.0, -1.0, 2.0, np.pi/3, -np.pi/4, np.pi/6]),
    np.array([-2.0, 3.5, 1.0, -np.pi/2, np.pi/5, -np.pi/3]),
])
def test_se3_inverse(xi):
    X = _xi_to_tuple(xi)
    t_inv_expected = -X[1].T @ X[0]
    R_inv_expected = X[1].T
    t_inv_result, R_inv_result = se3_inverse(X)
    np.testing.assert_array_almost_equal(t_inv_result, t_inv_expected)
    np.testing.assert_array_almost_equal(R_inv_result, R_inv_expected)


@pytest.mark.parametrize("xi", [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([1.0, -2.0, 0.5, np.pi/6, 0.0, 0.0]),
    np.array([-1.0, 1.5, 2.0, 0.0, np.pi/4, -np.pi/3]),
])
def test_se3_compose_with_inverse(xi):
    X = _xi_to_tuple(xi)
    X_inv = se3_inverse(X)
    t_result, R_result = se3_compose(X, X_inv)
    np.testing.assert_array_almost_equal(t_result, np.zeros(3))
    np.testing.assert_array_almost_equal(R_result, np.eye(3))


@pytest.mark.parametrize("xi,expected", [
    (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros((4, 4))),
    (np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]), np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
    ])),
    (np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]), np.array([
        [0.0, -3.0, 2.0, 0.0],
        [3.0, 0.0, -1.0, 0.0],
        [-2.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])),
])
def test_se3_wedge(xi, expected):
    np.testing.assert_array_almost_equal(se3_wedge(xi), expected)


@pytest.mark.parametrize("xi", [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([1.0, -2.0, 3.0, 0.5, -0.3, 0.7]),
    np.array([-3.0, 4.0, -1.0, -1.2, 0.8, -0.5]),
])
def test_se3_vee_wedge_roundtrip(xi):
    np.testing.assert_array_almost_equal(se3_vee(se3_wedge(xi)), xi)


@pytest.mark.parametrize("xi", [
    np.array([1.5, -2.0, 3.0, 0.0, 0.0, 0.0]),
    np.array([0.5, 1.0, -1.0, 0.0, 0.0, 0.0]),
])
def test_se3_exp_zero_rotation(xi):
    result = se3_exp(xi)
    expected = np.eye(4)
    expected[0, 3] = xi[0]
    expected[1, 3] = xi[1]
    expected[2, 3] = xi[2]
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("xi", [
    np.array([1.0, -0.5, 0.7, 1e-10, 0.0, 0.0]),
    np.array([-2.0, 3.0, 1.0, 0.0, -1e-9, 1e-9]),
    np.array([0.3, -1.2, 0.4, 1e-7, 1e-7, -1e-7]),
])
def test_se3_exp_near_zero_rotation(xi):
    X = se3_exp(xi)
    xi_back = se3_log(X)
    np.testing.assert_array_almost_equal(xi_back, xi, decimal=7)


@pytest.mark.parametrize("xi", [
    np.array([1.0, 2.0, 3.0, 0.2, -0.3, 0.4]),
    np.array([-1.5, 0.4, -0.8, -0.7, 0.5, 1.0]),
    np.array([0.0, -2.2, 1.5, 1.3, -0.6, 0.1]),
])
def test_se3_exp_blocks(xi):
    phi = np.array([xi[3], xi[4], xi[5]])
    result = se3_exp(xi)
    np.testing.assert_array_almost_equal(result[0:3, 0:3], so3_exp(phi))
    np.testing.assert_array_almost_equal(result[3, :], np.array([0.0, 0.0, 0.0, 1.0]))


def test_se3_exp_log_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        xi = np.concatenate([
            rng.uniform(-2.0, 2.0, size=3),
            _random_rotvec(rng),
        ])
        X = se3_exp(xi)
        np.testing.assert_array_almost_equal(se3_exp(se3_log(X)), X, decimal=7)


def test_se3_log_exp_random():
    rng = np.random.default_rng(1)
    for _ in range(200):
        xi = np.concatenate([
            rng.uniform(-2.0, 2.0, size=3),
            _random_rotvec(rng),
        ])
        np.testing.assert_array_almost_equal(se3_log(se3_exp(xi)), xi, decimal=7)


def _random_rotvec(rng, max_norm=np.pi - 1e-3):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, max_norm)
    return axis * angle


def test_se3_adjoint_composition_random():
    rng = np.random.default_rng(2)
    for _ in range(50):
        xi1 = np.concatenate([rng.uniform(-2.0, 2.0, size=3), _random_rotvec(rng)])
        xi2 = np.concatenate([rng.uniform(-2.0, 2.0, size=3), _random_rotvec(rng)])
        X1 = _xi_to_tuple(xi1)
        X2 = _xi_to_tuple(xi2)
        t12, R12 = se3_compose(X1, X2)
        phi12 = so3_log(R12)
        xi12 = np.concatenate([t12, phi12])
        Ad_12 = se3_Ad(xi12)
        Ad_1 = se3_Ad(xi1)
        Ad_2 = se3_Ad(xi2)
        np.testing.assert_array_almost_equal(Ad_12, Ad_1 @ Ad_2, decimal=7)


def test_se3_adjoint_inverse_random():
    rng = np.random.default_rng(3)
    for _ in range(50):
        xi = np.concatenate([rng.uniform(-2.0, 2.0, size=3), _random_rotvec(rng)])
        X = _xi_to_tuple(xi)
        t_inv, R_inv = se3_inverse(X)
        phi_inv = so3_log(R_inv)
        xi_inv = np.concatenate([t_inv, phi_inv])
        Ad_inv = se3_Ad(xi_inv)
        Ad = se3_Ad(xi)
        np.testing.assert_array_almost_equal(Ad_inv, np.linalg.inv(Ad), decimal=7)


def test_se3_adjoint_definition_conjugation_random():
    rng = np.random.default_rng(4)
    for _ in range(30):
        xi_X = np.concatenate([rng.uniform(-2.0, 2.0, size=3), _random_rotvec(rng)])
        eta = np.concatenate([rng.uniform(-2.0, 2.0, size=3), rng.uniform(-2.0, 2.0, size=3)])
        X_tuple = _xi_to_tuple(xi_X)
        X_mat = _tuple_to_matrix(X_tuple)

        lhs = se3_vee(X_mat @ se3_wedge(eta) @ np.linalg.inv(X_mat))
        rhs = se3_Ad(xi_X) @ eta
        np.testing.assert_array_almost_equal(lhs, rhs, decimal=7)
